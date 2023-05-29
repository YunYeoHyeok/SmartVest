from dataclasses import dataclass


@dataclass
class DBinfo:
    host: str
    user: str
    password: str
    database: str
    charset: str


db_info = DBinfo(
    host="localhost",
    user="root",
    password="pjo7171",
    database="securevest",
    charset="utf8",
)
