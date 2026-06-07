#!/bin/bash
# Download the public datasets FlashLOB needs. Both are free.
set -e

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
echo "Data dir: $DATA_DIR"

# ---------------------------------------------------------------------------
# 1) FI-2010 benchmark (for Table 3 cross-model comparison).
#    Source: Ntakaris et al., hosted on the Finnish IDA / etsin repository.
#    Use the NoAuction / DecPre (decimal precision) variant to match protocol.py.
# ---------------------------------------------------------------------------
echo ""
echo "=== FI-2010 ==="
echo "Manual step (the host requires accepting terms / a browser download):"
echo "  1. Open: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649"
echo "  2. Download 'BenchmarkDatasets.zip' and unzip it."
echo "  3. Copy the folder 'NoAuction/3.NoAuction_DecPre/' into:"
echo "       $DATA_DIR/fi2010/"
echo "  Expected files include Train_Dst_NoAuction_DecPre_CF_7.txt and Test_..._CF_7.txt"
echo ""

# ---------------------------------------------------------------------------
# 2) LOBSTER free sample (for Phase -1/0/1 long-context experiments).
#    LOBSTER publishes free sample message+orderbook files (e.g. AAPL, level 10).
# ---------------------------------------------------------------------------
echo "=== LOBSTER sample ==="
echo "Manual step (LOBSTER sample page, free):"
echo "  1. Open: https://lobsterdata.com/info/DataSamples.php"
echo "  2. Download a level-10 sample, e.g. AAPL_2012-06-21_34200000_57600000_message_10.csv"
echo "     and the matching _orderbook_10.csv"
echo "  3. Place both in: $DATA_DIR/lobster/"
echo "  4. Point experiments/configs/base.yaml data.lobster_message / data.lobster_book at them."
echo ""
echo "Tip: until you have real data, generate a synthetic fixture for smoke tests:"
echo "  python scripts/make_synthetic_lobster.py --out $DATA_DIR/lobster --n 20000"
