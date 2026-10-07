import json, time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import pandas as pd

UA="Mozilla/5.0 (compatible; NASCAR-Predictor-Source-Diagnostic/2.4)"

TESTS=[
    ("NASCAR Feed — Drivers","https://feed.nascar.com/api/Driver"),
    ("NASCAR Feed — Driver Summary","https://feed.nascar.com/api/DriverSummary"),
    ("NASCAR Feed — Season Stats","https://feed.nascar.com/api/stats/season"),
    ("NASCAR Reference — Home","https://nascar-reference.com/"),
    ("NASCAR Reference — Drivers","https://nascar-reference.com/drivers"),
    ("NASCAR Reference — 2026 Results","https://nascar-reference.com/results"),
    ("NASCAR Reference — Entry Lists","https://nascar-reference.com/entry-lists"),
]

def probe(name,url,timeout=10):
    started=time.time()
    req=Request(url,headers={"User-Agent":UA,"Accept":"application/json,text/html,*/*"})
    try:
        with urlopen(req,timeout=timeout) as r:
            body=r.read(150000)
            ctype=r.headers.get("content-type","")
            status=getattr(r,"status",200)
        preview=body[:160].decode("utf-8","replace").replace("\n"," ")
        kind="JSON" if "json" in ctype.lower() or preview.lstrip().startswith(("{","[")) else "HTML/Text"
        return {"Source":name,"URL":url,"HTTP":status,"Reachable":True,"Type":kind,
                "Bytes sampled":len(body),"Seconds":round(time.time()-started,2),"Preview":preview}
    except HTTPError as e:
        return {"Source":name,"URL":url,"HTTP":e.code,"Reachable":False,"Type":"HTTP error",
                "Bytes sampled":0,"Seconds":round(time.time()-started,2),"Preview":str(e)}
    except Exception as e:
        return {"Source":name,"URL":url,"HTTP":None,"Reachable":False,"Type":"Connection error",
                "Bytes sampled":0,"Seconds":round(time.time()-started,2),"Preview":str(e)}

def run_source_diagnostic():
    return pd.DataFrame([probe(n,u) for n,u in TESTS])
