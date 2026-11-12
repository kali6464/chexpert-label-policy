#!/bin/bash
# Download NIH ChestX-ray14 (112,120 PNG images, about 45 GB) from the NIH Clinical Center's
# public Box links. No account needed. Run on a login node, from the folder that should hold it:
#   mkdir -p <external>/nih && cd <external>/nih && bash /path/to/download_nih.sh
# Safe to re-run: finished archives are skipped. Two files must be fetched by hand in a browser
# from https://nihcc.app.box.com/v/ChestXray-NIHCC and copied here:
#   Data_Entry_2017_v2020.csv   and   test_list.txt
set -e
URLS=(
https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz
https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz
https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz
https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz
https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz
https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz
https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz
https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz
https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz
https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz
https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz
https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz
)
for i in "${!URLS[@]}"; do
  n=$(printf "%02d" $((i + 1)))
  if [ -e "done_${n}" ]; then echo "archive ${n}: already extracted"; continue; fi
  echo "archive ${n}: downloading"
  wget -c -q --show-progress -O "images_${n}.tar.gz" "${URLS[$i]}"
  tar -xzf "images_${n}.tar.gz" && rm "images_${n}.tar.gz" && touch "done_${n}"
done
echo "PNG files: $(ls images | wc -l)   (expected 112120)"
ls Data_Entry_2017_v2020.csv test_list.txt 2>/dev/null || echo "Still needed from the Box page: Data_Entry_2017_v2020.csv and test_list.txt"
