# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Contains enumeration values used with :class:`azureml.core.dataset.Dataset`."""

from enum import Enum


class HistogramCompareMethod(Enum):
    """Defines metric for measuring the difference between distributions of numeric columns of two dataset profiles.

    These enumeration values are used in the Dataset class method
    :meth:`azureml.core.dataset.Dataset.compare_profiles`.
    """

    WASSERSTEIN = 0  #: Selects Wasserstein distance metric
    ENERGY = 1  #: Selects Energy distance metric


class PromoteHeadersBehavior(Enum):
    """Defines options for how column headers are processed when reading data from files to create a dataset.

    These enumeration values are used in the Dataset class method
    :meth:`azureml.core.dataset.Dataset.from_delimited_files`.
    """

    NO_HEADERS = 0  #: No column headers are read
    ONLY_FIRST_FILE_HAS_HEADERS = 1  #: Read headers only from first row of first file, everything else is data.
    COMBINE_ALL_FILES_HEADERS = 2  #: Read headers from first row of each file, combining identically named columns.
    ALL_FILES_HAVE_SAME_HEADERS = 3  #: Read headers from first row of first file, drops first row from other files.


class SkipLinesBehavior(Enum):
    """Defines options for how leading rows are processed when reading data from files to create a dataset.

    These enumeration values are used in the Dataset class method
    :meth:`azureml.core.dataset.Dataset.from_delimited_files`.
    """

    NO_ROWS = 0  #: All rows from all files are read, none are skipped.
    FROM_FIRST_FILE_ONLY = 1  #: Skip rows from  first file, reads all rows from other files.
    FROM_ALL_FILES = 2  #: Skip rows from each file.


class FileEncoding(Enum):
    """Defines file encoding options used when reading data from files to create a dataset.

    These enumeration values are used in the Dataset class methods which load data from files and
    the encoding can be specified, like in the method
    :meth:`azureml.core.dataset.Dataset.from_delimited_files`.
    """

    UTF8 = 0
    ISO88591 = 1
    LATIN1 = 2
    ASCII = 3
    UTF16 = 4
    UTF32 = 5
    UTF8BOM = 6
    WINDOWS1252 = 7


class FileType(Enum):
    """DEPRECATED. Use strings instead."""

    import warnings
    warnings.warn(
        'FileType Enum is Deprecated in > 1.0.39. Use strings instead.',
        category=DeprecationWarning)

    GENERIC_CSV = 'GenericCSV'
    GENERIC_CSV_NO_HEADER = 'GenericCSVNoHeader'
    GENERIC_TSV = 'GenericTSV'
    GENERIC_TSV_NO_HEADER = 'GenericTSVNoHeader'
    ZIP = 'Zip'
    UNKNOWN = 'Unknown'
