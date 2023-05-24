import sys
import json
import argparse
def formater(string,sort_key=True,indent=4):
    loaded_json= json.loads(string)
    return json.dumps(loaded_json,sort_keys=sort_key,indent=indent)

def main (path,no_sort):
    if no_sort:
        sort_key=True
    else: sort_key=False

    with open (path,'r') as _:
        print (
            formater(_.read(),sort_key=True)
              )
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="this is a json formater method")
    parser.add_argument("--sort",action="store_true",help="set sorting")
    args=parser.parse_args()
    print(args.sort)

    #main(sys.argv[-1],no_sort=False)