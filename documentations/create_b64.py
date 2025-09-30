import base64, os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AnomalyDetection_Detailed.docx')
out = p + '.b64'
with open(p, 'rb') as f, open(out, 'wb') as w:
    data = f.read()
    b = base64.b64encode(data)
    w.write(b)
print('Wrote', out, 'size_bytes=', os.path.getsize(out))

