from data_processor import DataProcessor  # Replace 'your_module' with the actual module name where DataProcessor is defined
from config import Config  # Make sure to replace 'config' with the actual module name if different

def test_process_eia_data():
    processor = DataProcessor(Config.load())
    raw_data = {
        'response': {
            'data': [
                {'period': '2025-04-25', 'respondent': 'SCL', 'value': 1000},
                {'period': '2025-04-26', 'respondent': 'SCL', 'value': 1100}
            ]
        }
    }
    df = processor.process_eia_data(raw_data, 'SCL')
    assert df is not None
    assert list(df.columns) == ['date', 'energy_demand', 'eia_region', 'data_source', 'processed_at']
    assert len(df) == 2