id=1ugB5wHFL3WMqW_c1OIPJJAZs8_RN-PCF
filename=cnn-fold1-1602524723_25-1.00.tar

echo "Creating results directory..."
mkdir -p results; cd results

echo "Downloading checkpoint ${filename}..."
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${id}" -o ${filename}
