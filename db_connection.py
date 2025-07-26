import psycopg2


host = "pg-2bf10469-calcmate.c.aivencloud.com"          
database = "defaultdb"  
user = "avnadmin"        
password = "AVNS_wWJ2o9wNyuyl6K6Ww5u"  
port = "21820"              

conn=psycopg2.connect(
    host=host,
    database=database,
    user=user,
    password=password,
    port=port
)