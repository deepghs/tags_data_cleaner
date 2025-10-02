import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
from hbutils.string import plural_word
from tqdm import tqdm


@dataclass
class TagData:
    df_table: pd.DataFrame
    df_tags: pd.DataFrame


def load_from_parquet(
        df_table_file: str,
        df_tags_file: Optional[str] = None,
        table_id_column: str = 'id',
        table_tags_column: str = 'tags',
        table_tags_preprocess: Callable = lambda x: x,
        tags_id_column: str = 'name',
        batch_size: int = 100000,
) -> TagData:
    logging.info(f'Loading table from {df_table_file!r} ...')
    df_raw_table = pd.read_parquet(df_table_file)

    if df_tags_file:
        logging.info(f'Loading tags from {df_tags_file!r} ...')
        df_raw_tags = pd.read_parquet(df_tags_file)
    else:
        logging.info('No tags table given, will be auto-calculated later.')
        df_raw_tags = None

    if table_id_column not in df_raw_table:
        raise RuntimeError(f'Table ID column {table_id_column!r} not found in table.')
    if table_tags_column not in df_raw_table:
        raise RuntimeError(f'Table tags column {table_tags_column!r} not found in table.')

    if df_raw_tags is not None and tags_id_column not in df_raw_tags:
        raise RuntimeError(f'Tags ID column {tags_id_column!r} not found in tags.')
    if df_raw_tags is not None:
        expected_tags = set(df_raw_tags[tags_id_column].tolist())
        logging.info(f'{plural_word(len(expected_tags), "expected tag")} checked in.')
    else:
        logging.info('All tags will be included due to the tag list file is not given.')
        expected_tags = None

    d_tags = defaultdict(lambda: 0)
    table = []
    for _, item in tqdm(df_raw_table.iterrows(), total=len(df_raw_table), desc='Scan Table'):
        d_item = item.to_dict()
        id_, tags = d_item[table_id_column], table_tags_preprocess(d_item[table_tags_column])
        if expected_tags is not None:
            tags = [tag for tag in tags if tag in expected_tags]
        table.append({'id': id_, 'tags': tags})
        for tag in tags:
            d_tags[tag] += 1

    df_table = pd.DataFrame(table)
    df_table = df_table.sort_values(by=['id'], ascending=[False])
    df_tags = pd.DataFrame([{'name': tag, 'count': d_tags[tag]} for tag in d_tags])
    df_tags = df_tags.sort_values(by=['count', 'name'], ascending=[False, True])
    return TagData(
        df_table=df_table,
        df_tags=df_tags,
    )
