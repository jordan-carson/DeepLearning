import pandas as pd


def df_astype(df, types):
    """
    Changes data types of pandas DataFrame according to provided dictionary as older versions of
    pandas do not support df.astype(dict)
    :param df: pandas DataFrame
    :param types: Dictionary with (column: dtype) to be applied to DataFrame
    :return: DataFrame with modified types
    """
    if not isinstance(types, dict):
        raise ValueError('Expected {} as argument of {}, got={}'.format(dict.__name__, __name__, type(types)))

    for col, dtype in types.items():
        df[col] = df[col].astype(dtype)

    return df


def df_apply(df, funcs):
    """
    Applys function(s) to DataFrame
    :param df: pandas DataFrame
    :param funcs: callable function or dict containing column: list of callable functions
    :return: resulting DataFrame
    """
    if not callable(funcs) and not isinstance(funcs, dict):
        raise ValueError('Expected {} as argument of {}, got={}'.format(
            ', '.join([a.__name__ for a in [callable, dict]]), __name__, type(funcs)
        ))

    if callable(funcs):
        return df.apply(funcs, axis=1)

    for column, operations in funcs.items():
        for operation in operations:
            df[column] = df.apply(operation, axis=1)
    return df



