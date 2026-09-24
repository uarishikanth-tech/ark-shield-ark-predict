from dataclasses import dataclass
from typing import Any, Sequence

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100


@dataclass
class Pagination:
    page: int
    page_size: int

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        return self.page_size


def parse_pagination(page: int | None, page_size: int | None) -> Pagination:
    p = page if page and page >= 1 else 1
    ps = page_size if page_size and page_size >= 1 else DEFAULT_PAGE_SIZE
    ps = min(ps, MAX_PAGE_SIZE)
    return Pagination(page=p, page_size=ps)


def build_paginated_result(data: Sequence[Any], total: int, pagination: Pagination) -> dict:
    return {
        "data": data,
        "page": pagination.page,
        "pageSize": pagination.page_size,
        "total": total,
        "totalPages": max(1, -(-total // pagination.page_size)),  # ceil division
    }
