from typing import Any, List


def dump_params(m: Any, with_name: bool = False) -> None:
    if with_name:
        for p in m.named_parameters():
            print(f"{p[0]}: \n\t{p[1].shape}")
    else:
        for p in m.parameters():
            print(p.shape)


def matrix_size(dims: List[int], show_details: bool = False) -> int:
    n = dims[0]
    for d in dims[1:]:
        n *= d
    if show_details:
        print(dims, " -> ", n)
    return n


def total_params(m: Any, show_details: bool = False) -> int:
    return sum([matrix_size(p.size(), show_details) for p in m.parameters()])
