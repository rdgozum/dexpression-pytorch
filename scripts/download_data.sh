ids=(1mzw-ZMvBicBc_KNyYCM5FQ4gH06zbMlc 1Hs1qZ0nBFC6pEUAnMfROT6VfeS1G_4iG)
filenames=(x.py y.py)

echo "Creating data directory..."
mkdir -p data; cd data

for i in ${!ids[@]};
do
  id=${ids[$i]}
  filename=${filenames[$i]}

  echo ""
  echo "Downloading data ${filename}..."
  curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${id}" > /dev/null
  code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
  curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${id}" -o ${filename}
done
