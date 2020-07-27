import re
import sqlite3
import sys


def main(file, replacement):
    conn = sqlite3.connect(file)
    cur = conn.cursor()
    cur.execute("select path, id from file")

    patched = [
        (re.sub(r"/.*/RAiDER/", replacement, r[0]), r[1])
        for r in cur
    ]

    cur.executemany("update file set path=? where id=?", patched)

    conn.commit()
    conn.close()


if __name__ == '__main__':
    main(*sys.argv[1:])
