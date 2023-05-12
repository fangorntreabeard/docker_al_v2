from time import time
from typing import Any

import pg8000 as pg
from contextlib import closing

from pydantic import BaseModel


class Stage(BaseModel):
    stage_type: str
    stage_num: int
    num_of_stages: int


class ResultModel(BaseModel):
    data: Any
    message: str
    stage: str | Stage


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Connection(metaclass=Singleton):
    def __init__(
            self,
            host: str,
            dbname: str,
            user: str,
            password: str,
            port: str = 5432,
    ):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def _execute(self, _q: str):
        with closing(pg.connect(
                    database=self.dbname,
                    user=self.user,
                    password=self.password,
                    host=self.host,
                    port=self.port,
        )) as conn:
            cursor = conn.cursor()
            cursor.execute(_q)
            conn.commit()


class ResultSetter:
    def __init__(self, result_id: str, result: str):

        self.result = result
        self.result_id = result_id

    def intermediate(self):
        return Connection(...)._execute(
            f"""
                    UPDATE public."result"
                    SET "result"='{self.result}'
                    WHERE "ID"='{self.result_id}';
            """
        )

    def final(self):
        return Connection(...)._execute(
            f"""
                    UPDATE public."result"
                    SET "result"='{self.result}', time_end={time()}
                    WHERE "ID"='{self.result_id}';
            """
        )


class Result(ResultModel):
    def set(self, result_id: str):
        return ResultSetter(result_id=result_id, result=self.json())


def register(
        host: str,
        dbname: str,
        user: str,
        password: str,
):
    return Connection(
        host=host,
        dbname=dbname,
        user=user,
        password=password,
    )


if __name__ == '__main___':
    register(
        'some_host',
        'some_db',
        user='user',
        password='1234'
    )
    Result(msg='done', data='', progress='1/1').set('mCXX0aaIzhOX6kBOQ6XQ').final()
