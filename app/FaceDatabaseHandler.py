import sqlite3
import numpy as np
import io

class FaceDatabaseHandler:
    def __init__(self, db_name='data/faces.db'):
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, self._adapt_numpy_array)
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("nparray", self._convert_numpy_array)

        # connect to database
        self.connection = sqlite3.connect(db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.connection.cursor()

        if not self._is_face_table_present():
            self._create_face_table()

    def __del__(self):
        self.connection.close()

    def _create_face_table(self):
        self.cursor.execute('CREATE TABLE faces (name text, embeddings nparray);')
    
    def _is_face_table_present(self):
        try:
            self.cursor.execute('SELECT * FROM faces LIMIT 1;')
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
        self.cursor.execute('SELECT name FROM faces;')
        names = self.cursor.fetchall()
        names = self._eliminate_tuples(names)
        names = list(set(names)) # keep unique elements only
        return names

    def get_person_embeddings(self, name):
        try:
            self.cursor.execute(f'SELECT embeddings FROM faces WHERE name=?;', name)
            embeddings = self.cursor.fetchall()
        except sqlite3.OperationalError:
            return []

        return self._eliminate_tuples(embeddings)

    def add_embedding(self, name, embedding):
        self.cursor.execute('INSERT INTO faces VALUES (?, ?);', (name, embedding))
        self.connection.commit()

    def delete_person(self, name):
        self.cursor.execute('DELETE FROM faces WHERE name=?;', [name])
        self.connection.commit()