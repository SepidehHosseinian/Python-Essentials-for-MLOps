# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from datetime import datetime, timedelta

SUPPORTED_VALUE_TYPE_TUPLE = (int, float, str, datetime, timedelta)
TABLE_FMT = '<table style="width:100%">{0}</table>'
ROW_FMT = "<tr>{0}</tr>"
HEADER_FMT = "<th>{0}</th>"
DATA_FMT = "<td>{0}</td>"
# target="_blank" opens in new tab, rel="noopener" is for perf + security
LINK_FMT = '<a href="{0}" target="_blank" rel="noopener">{1}</a>'
