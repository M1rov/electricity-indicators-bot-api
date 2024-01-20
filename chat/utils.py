from consts import months


def get_months_plural(count):
    if count == 1:
        return 'місяць'
    if count < 5:
        return 'місяці'
    return 'місяців'


def format_electricity_recording(row):
    return f"{months[row.month]} {row.year}: {row.kWh} кВг"
