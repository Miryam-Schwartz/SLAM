from VAN_ex.code.DB.DataBase import DataBase


def ex4_run():
    db = DataBase()
    db.fill_database(20)
    db.save_database('C:\\Users\\Miryam\\SLAM\\VAN_ex\\code\\DB\\')
    return 0

ex4_run()