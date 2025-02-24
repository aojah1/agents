# NL2SQL Demo
from langchain_community.utilities import SQLDatabase 
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool


# Define SQLite database file
DATABASE_URL = "sqlite:///Chinook.db"

## Create Sample Data
def initialize_db():

    # Create engine
    engine = create_engine(DATABASE_URL, echo=True)

    # Define base class
    Base = declarative_base()

    # Define Employees Table
    class Employees(Base):
        __tablename__ = 'Employees'

        EmployeeID = Column(Integer, primary_key=True, autoincrement=True)
        FirstName = Column(String, nullable=False)
        LastName = Column(String, nullable=False)
        Age = Column(Integer)
        Department = Column(String)

    # Create the table in the database
    Base.metadata.create_all(engine)

    # Create a session
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    # Insert sample data (if table is empty)
    if not session.query(Employees).first():
        sample_employees = [
            Employees(FirstName="Anup", LastName="Ojah", Age=30, Department="HR"),
            Employees(FirstName="Saurabh", LastName="Mishra", Age=25, Department="Finance"),
            Employees(FirstName="Dmitry", LastName="Baev", Age=40, Department="IT"),
        ]
        session.add_all(sample_employees)
        session.commit()

    # Query and display employees
    employees = session.query(Employees).all()
    for emp in employees:
        print(f"{emp.EmployeeID}: {emp.FirstName} {emp.LastName}, {emp.Age} years, {emp.Department}")

    # Close the session
    session.close()
    
if __name__ == "__main__":
    try:
        initialize_db()
        
    except Exception as e:
        print(f"Fatal error: {e}")
    
    
