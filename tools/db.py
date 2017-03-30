import os
import sqlite3
import time


class db:
    attr = []
    attr_type = []
    filename = ""
    table_name = "training_set"
    
    def __init__(self, filename, attr, attr_type):
        self.attr = attr
        self.attr_type = attr_type
        self.filename = filename
        
        if(len(self.attr) != len(self.attr_type)):
            raise Exception("Attribute length is not equal to attribute type length!")
        
        tmp_str = """Initialize database """ + self.filename + """ with attributes 
    """ + str(self.attr) + """ and types
    """ + str(self.attr_type)
                
        self.output(tmp_str)
                
    def output(self, msg):
        for line in msg.splitlines():
            print "DB: " + line
        
    def check_file(self):
        if(self.filename == ""):
            raise Exception("Filename is empty!")
            
        if(not(self.filename.endswith(".db"))):
            raise Exception("Not a database file format!")
        
        
    def sql_exec(self, c, command , debug):
        if(debug):
            self.output("""SQL-DEBUG: Executing statement: \nSQL-DEBUG: """ + command)
                        
        c.execute(command)
        
        if(debug):
            self.output("SQL-DEBUG: Done!")
    
    
    
    def write(self, tmp_tuple):
        self.check_file()
                
        # Delete previous db
        if(os.path.isfile(self.filename)):
            self.output("Database already exists. Deleting...")
            os.remove(self.filename)
            
        
        self.output("Write to database " + self.filename)
        
        # establish db connection
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        
        # rdy for commands
        command = """CREATE TABLE """ + self.table_name + """
                     ("""
        for i in range(len(self.attr) - 1):
            command += self.attr[i] + " " + self.attr_type[i] + ","
            
        command += self.attr[len(self.attr) - 1] + " " + self.attr_type[len(self.attr) - 1] + ")"
            
        
        self.sql_exec(c, command, False) 
        
        
        self.output("Fill in data")
        
        start = time.time()
        
        command = "INSERT INTO " + self.table_name + "('"
        for i in range(len(self.attr) - 1):
            command += self.attr[i] + "', '"
        
        command += self.attr[len(self.attr) - 1] + "')VALUES ("
        for i in range(len(self.attr) - 1):
            command += "?, "
            
        command += "?)"
        
        
        c.executemany(command, tmp_tuple)

        end = time.time()
        conn.commit()
        
        self.output("Done! Time needed: " + str(end - start))
        
        self.output("Close connection to database")
        conn.commit()
        conn.close()
        self.output("Writing done!")
        
        
        
    def read(self):
        self.check_file()
        if(not(os.path.isfile(self.filename))):
            raise Exception("Cannot reach file!")
        self.output("Reading file " + self.filename)
        
        # establish db connection
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        
        command = "SELECT * FROM " + self.table_name
        
        self.sql_exec(c, command, False)
            
        start = time.time()
        data = c.fetchall()
        end = time.time()
        

        conn.commit()
        conn.close()
        self.output("Reading done! Time needed: " + str(end - start))
        
        return data
        
    def read_input(self):
        self.check_file()
        if(not(os.path.isfile(self.filename))):
            raise Exception("Cannot reach file!")
        self.output("Reading file " + self.filename)
        
        # establish db connection
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        
        command = "SELECT "
        
        for i in range(len(self.attr) - 2):
            command += self.attr[i] + ", "
        
        command += self.attr[len(self.attr) - 2] + " FROM " + self.table_name
        
        self.sql_exec(c, command, True)
            
        start = time.time()
        data = c.fetchall()
        end = time.time()
        

        conn.commit()
        conn.close()
        self.output("Reading done! Time needed: " + str(end - start))
        
        return data
    
    
    def read_output(self):
        self.check_file()
        if(not(os.path.isfile(self.filename))):
            raise Exception("Cannot reach file!")
        self.output("Reading file " + self.filename)
        
        # establish db connection
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        
        command = "SELECT " + self.attr[-1] + " FROM " + self.table_name
        
        self.sql_exec(c, command, False)
            
        start = time.time()
        data = c.fetchall()
        end = time.time()
        

        conn.commit()
        conn.close()
        self.output("Reading done! Time needed: " + str(end - start))
        
        return data
    
