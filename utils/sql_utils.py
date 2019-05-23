import mysql.connector


def create_mysql_table():
    classes = ['neutral', 'happiness', 'surprise', 'sadness',
               'anger', 'disgust', 'fear', 'contempt']
    conn = mysql.connector.connect(
      host="localhost",
      user="admin",
      passwd="admin",
      database="facedb")

    c = conn.cursor()
    c.execute(
      "CREATE TABLE IF NOT EXISTS sentiment(name TEXT, gender TEXT, age REAL,"
      "{} REAL, {} REAL, {} REAL, {} REAL, {} REAL,"
      "{} REAL, {} REAL, {} REAL, time TEXT)".format(*classes))
    conn.commit()

    conn.close()


def insert_mysql_table(data_list):
    conn = mysql.connector.connect(
      host="localhost",
      user="admin",
      passwd="admin",
      database="facedb")

    c = conn.cursor()

    for data in data_list:
        data = data[:11] + [data[-1]]
        c.execute(
          "INSERT INTO sentiment (name, gender, age,"
          "neutral, happiness, surprise, sadness, anger, "
          "disgust, fear, contempt, time) "
          "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
          [str(x) for x in data])
        conn.commit()

    conn.close()


def insert_mysql_ranktable(data_list):
    conn = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="skynet123$",
      database="deepai")
    c = conn.cursor()

    for data in data_list:
        c.execute("INSERT INTO ranktable (name, mainsentiment, score, photo) "
                  "VALUES (%s,%s,%s,%s)", [str(x) for x in data])
        conn.commit()

    conn.close()


def create_mysql_ranktable():
    conn = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="skynet123$",
      database="deepai")

    c = conn.cursor()
    c.execute(
      "CREATE TABLE IF NOT EXISTS ranktable("
      "name TEXT, mainsentiment TEXT, score REAL, photo TEXT)")
    conn.commit()

    conn.close()


def clean_mysql_table(table_name):
    conn = mysql.connector.connect(
      host="localhost",
      user="admin",
      passwd="admin",
      database="facedb")

    c = conn.cursor()
    c.execute("DELETE FROM {}".format(table_name))
    conn.commit()

    conn.close()
