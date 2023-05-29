from db_data import db_info
import mysql.connector


def db_connect(db_info):
    db = mysql.connector.connect(
        host=db_info.host,
        user=db_info.user,
        password=db_info.password,
        database=db_info.database,
    )
    return db
