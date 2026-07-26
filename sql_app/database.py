from sqlalchemy import StaticPool, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./prompt_app.db"
#SQLALCHEMY_DATABASE_URL = "mysql+mysqlconnector://root:root1234@localhost/fastapi_new"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=5, 
    max_overflow=0,
    pool_pre_ping=True, 
    isolation_level="READ UNCOMMITTED"
    # poolclass=StaticPool
    # , connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()