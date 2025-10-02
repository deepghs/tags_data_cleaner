import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, List

import pandas as pd
from hbutils.string import plural_word
from tqdm import tqdm


@dataclass
class TagData:
    df_table: pd.DataFrame
    df_tags: pd.DataFrame

    @classmethod
    def from_parquet(
            cls,
            df_table_file: str,
            df_tags_file: Optional[str] = None,
            table_id_column: str = 'id',
            table_tags_column: str = 'tags',
            table_tags_preprocess: Callable = lambda x: x,
            tags_id_column: str = 'name',
            tags_preserved_columns: Optional[List[str]] = None,
    ) -> 'TagData':
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

        tags_preserved_columns = list(tags_preserved_columns or [])
        if df_raw_tags is not None and tags_id_column not in df_raw_tags:
            raise RuntimeError(f'Tags ID column {tags_id_column!r} not found in tags.')
        if df_raw_tags is not None:
            expected_tags = set(df_raw_tags[tags_id_column].tolist())
            logging.info(f'{plural_word(len(expected_tags), "expected tag")} checked in.')
            d_raw_tags = {item[tags_id_column]: item for item in df_raw_tags.to_dict('records')}
        else:
            logging.info('All tags will be included due to the tag list file is not given.')
            expected_tags = None
            d_raw_tags = None

        d_tags = defaultdict(lambda: 0)
        table = []
        logging.info('Scanning table ...')
        for _, item in tqdm(df_raw_table.iterrows(), total=len(df_raw_table), desc='Scan Table'):
            d_item = item.to_dict()
            id_, tags = d_item[table_id_column], table_tags_preprocess(d_item[table_tags_column])
            if expected_tags is not None:
                tags = [tag for tag in tags if tag in expected_tags]
            table.append({'id': id_, 'tags': tags})
            for tag in tags:
                d_tags[tag] += 1

        logging.info('Making table ...')
        df_table = pd.DataFrame(table)
        df_table = df_table.sort_values(by=['id'], ascending=[False])
        df_tags = pd.DataFrame([{
            'name': tag,
            **{column: d_raw_tags[tag][column] for column in tags_preserved_columns},
            'count': d_tags[tag],
        } for tag in d_tags])
        df_tags = df_tags.sort_values(by=['count', 'name'], ascending=[False, True])
        logging.info(f'{plural_word(len(df_table), "sample")} in total:\n{df_table}')
        logging.info(f'{plural_word(len(df_tags), "tag")} in total:\n{df_tags}')
        return cls(
            df_table=df_table,
            df_tags=df_tags,
        )

    def recalculate_tags(self) -> 'TagData':
        logging.info('Recalculating the tags count by scanning the full table ...')
        d_tags = {item['name']: item for item in self.df_tags.to_dict('records')}
        d_tags_count = defaultdict(lambda: 0)
        for _, item in tqdm(self.df_table.iterrows(), total=len(self.df_table), desc='Scan Table'):
            d_item = item.to_dict()
            for tag in d_item['tags']:
                d_tags_count[tag] += 1

        logging.info('Remaking the tags table ...')
        df_tags = pd.DataFrame([{
            'name': tag,
            **{column: d_tags[tag][column] for column in self.df_tags.columns if column not in ['name', 'count']},
            'count': d_tags_count[tag],
        } for tag in d_tags_count])
        df_tags = df_tags.sort_values(by=['count', 'name'], ascending=[False, True])
        self.df_tags = df_tags
        logging.info(f'{plural_word(len(df_tags), "tag")} in total:\n{df_tags}')
        return TagData(
            df_table=self.df_table,
            df_tags=df_tags,
        )

    def clean_tags_in_table(self, recalculate_tags: bool = True) -> 'TagData':
        existing_tags = set(self.df_tags['name'])
        table = []
        logging.info('Scanning table for tags cleaning ...')
        for _, item in tqdm(self.df_table.iterrows(), total=len(self.df_table), desc='Scan Table'):
            d_item = item.to_dict()
            tags = [tag for tag in d_item['tags'] if tag in existing_tags]
            table.append({**d_item, 'tags': tags})

        logging.info('Re-making table ...')
        df_table = pd.DataFrame(table)
        df_table = df_table.sort_values(by=['id'], ascending=[False])
        logging.info(f'{plural_word(len(df_table), "sample")} in total:\n{df_table}')
        retval = TagData(
            df_table=df_table,
            df_tags=self.df_tags,
        )
        if recalculate_tags:
            retval = retval.recalculate_tags()
        return retval
