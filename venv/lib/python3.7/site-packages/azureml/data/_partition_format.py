# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains helper methods for partition format."""

import re
from azureml.data._dataprep_helper import dataprep


_date_parts = ['yyyy', 'MM', 'dd', 'HH', 'mm', 'ss']
_date_part_map = {d: '_sys_{}'.format(d) for d in _date_parts}
_path_record_key = '_sys_path_record'

validation_error = {
    'NO_PART': 'There must be at least one {column_name}',
    'NO_NAME': 'Column name is not specified.',
    'NO_YEAR': 'Format \'yyyy\' must be defined for datetime column.',
    'NO_DATE_FORMAT': 'Date format is not specified.',
    'BAD_BRACKET': 'Brackets must is pair and cannot be nested',
    'TOO_MANY_DATE': 'There cannot be more than one date part.',
    'RESERVED': 'The specified column name is reserved. Please use a different name.'
}


def handle_partition_format(dataflow, partition_format):
    validate_partition_format(partition_format)
    pattern, defined_date_parts, columns = parse_partition_format(partition_format)

    RegEx = dataprep().api.functions.RegEx
    col = dataprep().api.expressions.col
    create_datetime = dataprep().api.functions.create_datetime

    dataflow = dataflow.add_column(RegEx(pattern).extract_record(col('Path')), _path_record_key, None)

    for i in range(len(columns)):
        column = columns[i]
        if defined_date_parts and i == len(columns) - 1:
            parts = [col(_date_part_map[part], col(_path_record_key)) for part in defined_date_parts]
            exp = create_datetime(*parts)
        else:
            exp = col(column, col(_path_record_key))
        dataflow = dataflow.add_column(exp, column, None)

    dataflow = dataflow.drop_columns(_path_record_key)
    return dataflow


def validate_partition_format(partition_format):
    format_pattern = re.compile(r'{[^{|}]*}')
    escape_pattern = re.compile(r'{{|}}')
    no_escape = re.sub(escape_pattern, '', partition_format)
    parts = format_pattern.findall(no_escape)
    others = re.sub(format_pattern, '', no_escape)

    if len(parts) == 0:
        _raise_error(partition_format, 'NO_PART')

    if '{' in others or '}' in others:
        _raise_error(partition_format, 'BAD_BRACKET')

    has_date = False
    reserved_parts = ['{' + v + '}' for v in list(_date_part_map.values()) + [_path_record_key]]
    for part in parts:
        if part == '{}':
            _raise_part_error(part, partition_format, 'NO_NAME')
        if ':' in part:
            if has_date:
                _raise_part_error(part, partition_format, 'TOO_MANY_DATE')
            has_date = True
        if ':}' in part:
            _raise_part_error(part, partition_format, 'NO_DATE_FORMAT')
        if part in reserved_parts:
            _raise_part_error(part, partition_format, 'RESERVED')


def parse_partition_format(partition_format):
    defined_date_parts = []
    date_column = None
    columns = []
    i = 0
    pattern = ''
    while i < len(partition_format):
        c = partition_format[i]
        if c == '/':
            pattern += '\\/'
        elif partition_format[i:i + 2] in ['{{', '}}']:
            pattern += c
            i += 1
        elif c == '{':
            close = i + 1
            while close < len(partition_format) and partition_format[close] != '}':
                close += 1
            key = partition_format[i + 1:close]
            if ':' in key:
                date_column, date_format = key.split(':')
                for date_part in _date_parts:
                    date_format = date_format.replace(date_part, '{' + _date_part_map[date_part] + '}')
                partition_format = partition_format[:i] + date_format + partition_format[close + 1:]
                continue
            else:
                found_date = False
                for k, v in _date_part_map.items():
                    if partition_format.startswith(v, i + 1):
                        pattern_to_add = '(?<{}>\\d{{{}}})'.format(v, len(k))
                        if pattern_to_add in pattern:
                            pattern += '(\\d{{{}}})'.format(len(k))
                        else:
                            pattern += pattern_to_add
                            defined_date_parts.append(k)
                        found_date = True
                        break

                if not found_date:
                    pattern_to_add = '(?<{}>[^\\.\\/\\\\]+)'.format(key)
                    if pattern_to_add in pattern:
                        pattern += '([^\\.\\/\\\\]+)'
                    else:
                        columns.append(key)
                        pattern += pattern_to_add
                i = close
        elif c == '*':
            pattern += '(.*?)'
        elif c == '.':
            pattern += '\\.'
        else:
            pattern += c
        i += 1
    if date_column is not None:
        columns.append(date_column)

    if defined_date_parts and 'yyyy' not in defined_date_parts:
        raise _raise_error(partition_format, 'NO_YEAR')
    return pattern, defined_date_parts, columns


def _raise_error(partition_format, error):
    raise ValueError('Invalid partition_format "{}". {}'
                     .format(partition_format, validation_error[error]))


def _raise_part_error(part, partition_format, error):
    raise ValueError('Invalid part "{}" in partition_format "{}". {}'
                     .format(part, partition_format, validation_error[error]))
