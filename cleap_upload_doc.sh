#! /bin/bash
# script to upload documentation
echo "uploading documentation..."
rsync -r $2 /usr/local/share/doc/cleap-$1/html/ crinavar@dichato.dcc.uchile.cl:public_www/doc/cleap/
echo "done."
