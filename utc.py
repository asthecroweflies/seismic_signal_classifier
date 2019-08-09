# utility script for quick UTCDateTime conversions
from obspy import UTCDateTime

# e.g. 1545172970.90 -> 2018-12-18T22:42:50.900000Z
def ts2date(ts, print_date):
    date = UTCDateTime(float(ts))
    if (print_date):
        print(date)
    date = str(date)
    date = date.replace("-", "")
    date = date.replace("T", "")
    date = date.replace(":", "")
    date = date.replace("Z", "")
    date = date.replace(".", "")
    date = date[:18]
    return date

# e.g. "2018-12-18T22:42:50.900000Z" -> 1545172970.90
# or   "201812201940369389"          -> 1545334836.9389
# or   ""
def date2ts(UTCDate, print_ts):
    UTCDate = str(UTCDate)
    if ".dat" in UTCDate:
        UTCDate = UTCDate[:-4]
    if "vbox_" in UTCDate:
        UTCDate = UTCDate[5:]                                                   # removes "vbox_" prefix
        if "." not in UTCDate:
            UTCDate = UTCDate[:-4] + '.' + UTCDate[-4:]                         # inserts period after seconds quantity

    elif (len(UTCDate) == 18) and "." not in UTCDate:
        UTCDate = UTCDate[:-4] + '.' + UTCDate[-4:]                             # inserts period after seconds quantity

    ts = UTCDateTime(str(UTCDate)).timestamp
    if (print_ts):
        print(ts)
    return ts