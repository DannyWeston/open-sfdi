poetry export -f requirements.txt --output dist/requirements.txt
poetry run pip download -r dist/requirements.txt -d dist/wheels

echo
echo "Wheel strings:"
echo

for entry in "dist/wheels"/*
do
    filename=$(basename $entry)
    echo "\"wheels/$filename\","
done