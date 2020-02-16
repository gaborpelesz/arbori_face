import sqlite3
import numpy as np
import io

class FaceDatabaseHandler:
    def __init__(self):
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, self._adapt_numpy_array)
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("nparray", self._convert_numpy_array)

        # connect to database
        self.connection = sqlite3.connect('data/faces.db', detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.connection.cursor()

        if not self._is_name_table_present():
            self._create_name_table()

    def _create_name_table(self):
        self.cursor.execute('CREATE TABLE people (name text PRIMARY KEY);')
    
    def _is_name_table_present(self):
        try:
            self.cursor.execute('SELECT * FROM people LIMIT 1;')
        except sqlite3.OperationalError:
            return False
        return True

    def _adapt_numpy_array(self, arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def _convert_numpy_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    def _eliminate_tuples(self, database_list):
        # elements from database will be in form of ( element_value, __empty__) tuples 
        # with empty second value
        # this function eliminates unnecessary tuples with only 1 elements
        return list(map(lambda x: x[0], database_list))


    def get_people_names(self):
        self.cursor.execute('SELECT name FROM people;')
        names = self.cursor.fetchall()

        return self._eliminate_tuples(names)

    def get_person_embeddings(self, name):
        try:
            self.cursor.execute(f'SELECT embeddings FROM {name};')
            embeddings = self.cursor.fetchall()
        except sqlite3.OperationalError:
            return []

        return self._eliminate_tuples(embeddings)

    def add_embedding(self, name, embeddings):
        try:
            self.cursor.execute('INSERT INTO people VALUES (?);', [name])
            self.cursor.execute(f'CREATE TABLE {name} (embeddings nparray);')
        except sqlite3.IntegrityError:
            pass
        finally:
            self.cursor.execute(f'INSERT INTO {name} VALUES (?);', [embeddings])
            self.connection.commit()

    def delete_person(self, name):
        self.cursor.execute('DELETE FROM people WHERE name=?;', [name])
        self.cursor.execute(f'DROP TABLE {name};')
        self.connection.commit()