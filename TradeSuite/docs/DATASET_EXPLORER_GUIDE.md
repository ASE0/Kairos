# Dataset Explorer Window Guide

## Overview

The Dataset Explorer Window is a comprehensive tool for managing, analyzing, and visualizing trading datasets within the TradeSuite application. It provides a user-friendly interface for exploring datasets, applying filters, combining multiple datasets, and performing data analysis.

## Features

### 🔍 **Search and Filtering**
- **Text Search**: Search datasets by name, patterns, strategies, or any metadata
- **Quick Filters**: One-click filtering by acceptance status (Accepted, Rejected, Pending)
- **Advanced Filters**: 
  - Pattern-based filtering
  - Timeframe filtering
  - Location strategy filtering
  - Probability threshold filtering
  - Date range filtering

### 📊 **Data Visualization**
- **Price Charts**: View OHLC price data with high-low ranges
- **Volume Charts**: Analyze trading volume patterns
- **Distribution Analysis**: View returns distribution histograms
- **Correlation Matrix**: Analyze relationships between numeric columns
- **Time Range Selection**: Focus on specific time periods (30 days, 90 days, 6 months, 1 year)

### 📈 **Statistics Panel**
- Dataset overview (rows, columns, date range)
- Descriptive statistics for numeric columns
- Real-time updates when selecting different datasets

### 🔧 **Dataset Management**
- **Load Datasets**: Select and load datasets for use in other parts of the application
- **Combine Datasets**: Merge multiple datasets using AND, OR, or XOR logic
- **Accept/Reject**: Mark datasets as accepted or rejected with notes
- **Delete Datasets**: Remove unwanted datasets
- **Import/Export**: Support for CSV, Excel, and JSON formats

## Interface Layout

### Left Panel - Search & Filters
```
┌─────────────────────────┐
│    Search & Filter      │
├─────────────────────────┤
│ [Search Input Field]    │
│ [Accepted] [Rejected]   │
│ [Pending]               │
├─────────────────────────┤
│   Advanced Filters      │
├─────────────────────────┤
│ Status: [Dropdown]      │
│ Patterns: [List]        │
│ Timeframes: [List]      │
│ Location: [List]        │
│ Min Probability: [Spin] │
│ Date Filter: [Checkbox] │
│ [Clear All Filters]     │
└─────────────────────────┘
```

### Center Panel - Dataset List & Details
```
┌─────────────────────────────────────────┐
│           Dataset Table                 │
├─────────────────────────────────────────┤
│ Name | Status | Prob | Rows | Patterns │
│      |        |      |      |          │
├─────────────────────────────────────────┤
│         Dataset Details                 │
├─────────────────────────────────────────┤
│ [Detailed information about selected    │
│  dataset including metadata, patterns, │
│  strategies, and notes]                 │
├─────────────────────────────────────────┤
│ [Load] [Combine] [Accept] [Reject]     │
│ [Delete]                                │
└─────────────────────────────────────────┘
```

### Right Panel - Visualization & Statistics
```
┌─────────────────────────┐
│   Data Visualization    │
├─────────────────────────┤
│ Chart Type: [Dropdown]  │
│ Time Range: [Dropdown]  │
├─────────────────────────┤
│                         │
│      [Chart Area]       │
│                         │
├─────────────────────────┤
│      Statistics         │
├─────────────────────────┤
│ [Statistical summary    │
│  of selected dataset]   │
└─────────────────────────┘
```

## Usage Guide

### 1. Opening the Dataset Explorer
- From the main hub, click on "Dataset Explorer" in the menu
- The window will open and automatically load all available datasets

### 2. Searching for Datasets
- Use the search field to find datasets by name, patterns, or strategies
- Type keywords and the list will filter in real-time
- Use quotes for exact phrase matching

### 3. Applying Filters
- **Quick Filters**: Click the colored buttons (Accepted/Rejected/Pending) for instant filtering
- **Advanced Filters**: 
  - Select multiple patterns from the pattern list
  - Choose timeframes from the timeframe list
  - Set minimum probability thresholds
  - Enable date filtering to show only recent datasets

### 4. Viewing Dataset Details
- Click on any dataset row to view detailed information
- The details panel shows:
  - Dataset metadata (ID, creation date, status)
  - Data properties (row count, date range, probability)
  - Strategies and patterns used
  - Tags and notes
  - Parent datasets (if combined)

### 5. Data Visualization
- Select a dataset to enable visualization
- Choose chart type from the dropdown:
  - **Price Chart**: Shows OHLC data with high-low ranges
  - **Volume Chart**: Displays trading volume over time
  - **Distribution**: Histogram of price returns
  - **Correlation Matrix**: Relationships between numeric columns
- Adjust time range to focus on specific periods

### 6. Loading Datasets
- Select one or more datasets using the checkboxes
- Click "Load Selected" to load the first selected dataset
- The dataset will be available in other parts of the application

### 7. Combining Datasets
- Select 2 or more datasets using checkboxes
- Click "Combine Selected"
- Choose combination method:
  - **AND**: Intersection of datasets (common data points)
  - **OR**: Union of datasets (all data points)
  - **XOR**: Exclusive or (data points in odd number of datasets)
- Provide a name for the combined dataset
- The combined dataset will be saved and available for use

### 8. Managing Dataset Status
- Select a dataset and click "Accept Dataset" or "Reject Dataset"
- Optionally provide:
  - Probability value (0-100%)
  - Notes explaining the decision
- Status changes are saved and can be filtered

### 9. Importing External Data
- Click the "📥 Import" button in the toolbar
- Select a CSV, Excel, or JSON file
- Provide a name for the dataset
- The imported data will be saved and available for analysis

### 10. Exporting Datasets
- Select a dataset and click "📤 Export" in the toolbar
- Choose export format (CSV, Excel, JSON)
- Select save location
- The dataset will be exported with metadata

## Keyboard Shortcuts

- **Ctrl+F**: Focus search field
- **Ctrl+R**: Refresh dataset list
- **Delete**: Delete selected datasets (after confirmation)
- **Enter**: Load selected dataset
- **Escape**: Clear search/filters

## Tips and Best Practices

### Efficient Searching
- Use specific keywords for better results
- Combine multiple filters for precise filtering
- Use tags to organize datasets systematically

### Data Visualization
- Start with "All Data" time range to see the full picture
- Use specific time ranges to focus on recent trends
- Switch between chart types to understand different aspects of the data

### Dataset Management
- Regularly review and accept/reject datasets
- Use descriptive names and tags for easy identification
- Add notes to explain dataset characteristics or decisions

### Performance
- Large datasets may take time to load for visualization
- Use filters to reduce the number of datasets displayed
- Consider combining similar datasets to reduce clutter

## Troubleshooting

### Visualization Not Working
- Ensure matplotlib is installed: `pip install matplotlib`
- Check that the dataset has the required columns (Close, Volume, etc.)
- Try different chart types if one fails

### Import Issues
- Ensure the file format is supported (CSV, Excel, JSON)
- Check that the file has a proper date index
- Verify that numeric columns are properly formatted

### Performance Issues
- Close other applications to free up memory
- Use filters to reduce the number of displayed datasets
- Consider splitting very large datasets

## Integration with Other Components

The Dataset Explorer integrates with other parts of the TradeSuite:

- **Main Hub**: Datasets loaded here are available for backtesting
- **Strategy Builder**: Can use datasets for strategy development
- **Backtest Window**: Loaded datasets can be used for backtesting
- **Results Viewer**: Backtest results can be saved as datasets

## File Formats

### Supported Import Formats
- **CSV**: Comma-separated values with date index
- **Excel**: .xlsx files with date index
- **JSON**: JSON files with data and metadata

### Supported Export Formats
- **CSV**: Standard CSV format
- **Excel**: .xlsx format with formatting
- **JSON**: Complete dataset with metadata

## Data Structure

Datasets are stored with the following structure:
```
datasets/
├── data/
│   ├── [dataset_id].pkl.gz    # Compressed data files
├── metadata/
│   ├── [dataset_id].json      # Metadata files
└── index.json                 # Master index
```

## API Reference

### Signals
- `dataset_selected(dataset_id, data, info)`: Emitted when a dataset is selected
- `datasets_combined(dataset_id, data, probability)`: Emitted when datasets are combined

### Key Methods
- `refresh_datasets()`: Reload all datasets
- `load_dataset(dataset_id)`: Load specific dataset
- `combine_datasets(dataset_ids, method, name)`: Combine multiple datasets
- `export_dataset(dataset_id, path, format)`: Export dataset
- `import_dataset(filepath, name)`: Import external dataset

This comprehensive guide should help you make the most of the Dataset Explorer Window's powerful features for managing and analyzing your trading datasets. 