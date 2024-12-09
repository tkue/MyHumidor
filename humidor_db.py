from typing import List, Optional
from sqlalchemy import ForeignKey, String, TIMESTAMP, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy import engine


class BaseTable(DeclarativeBase):
    created_date = mapped_column(TIMESTAMP, server_default=func.now())
    updated_date = mapped_column(TIMESTAMP, server_default=func.now(), onupdate=func.current_timestamp)


class CigarManufacturer(BaseTable):
    __tablename__ = 'cigar_manufacturer'

    cigar_manufacturer_id: Mapped[int] = mapped_column(primary_key=True)
    cigar_manufacturer_name: Mapped[str] = mapped_column(String(length=50))

    cigar_product_lines: Mapped[List['CigarProductLine']] = relationship(
        back_populates='cigar_manufacturer', cascade='all, delete-orphan'
    )


class CigarProductLine(BaseTable):
    __tablename__ = 'cigar_product_line'

    cigar_product_line_id: Mapped[int] = mapped_column(primary_key=True)
    cigar_manufacturer_id: Mapped[int] = mapped_column(ForeignKey('cigar_manufacturer.cigar_manufacturer_id'))

    cigar_product_line_name: Mapped[str] = mapped_column(String(100))

    cigars: Mapped[List['Cigar']] = relationship(
        back_populates='cigar_product_line', cascade='all, delete-orphan'
    )

    cigar_manufacturer: Mapped['CigarManufacturer'] = relationship(back_populates='cigar_product_lines')


class Cigar(BaseTable):
    __tablename__ = 'cigar'

    cigar_id: Mapped[int] = mapped_column(primary_key=True)
    cigar_product_line_id: Mapped[int] = mapped_column(ForeignKey('cigar_product_line.cigar_product_line_id'))

    cigar_name: Mapped[str] = mapped_column(String(100))

    cigar_product_line: Mapped['CigarProductLine'] = relationship(back_populates='cigars')


def _get_engine_conn(database_name: str):
    return 'sqlite:///{}'.format(database_name)


def get_engine(database_name: str, echo: bool=False):
    conn_str = _get_engine_conn(database_name=database_name)
    return create_engine(conn_str, echo=echo)


def create_schema(engine: engine):
    BaseTable.metadata.create_all(engine)


# from sqlalchemy import create_engine
# engine = create_engine('sqlite:///humidor.db', echo=True)
# BaseTable.metadata.create_all(engine)

# from os import remove
# remove('humidor.db')

engine = get_engine(database_name='humidor.db', echo=True)
create_schema(engine=engine)


from sqlalchemy.orm import Session

with Session(engine) as session:
    padron = CigarManufacturer(
        cigar_manufacturer_name='Padron',
        cigar_product_lines=[
            CigarProductLine(cigar_product_line_name='1926 Anniversary'
                             , cigars=[
                                            Cigar(cigar_name='Belicoso')
                                        ])
        ]
    )

    cigars = [padron,]

    for cigar in cigars:
        session.merge(cigar)

    # session.add_all([padron])
    session.commit()



