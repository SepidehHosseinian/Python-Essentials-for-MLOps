import sys
import json
import argparse
import click
def formater(string,sort_key=True,indent=4):
    loaded_json= json.loads(string)
    return json.dumps(loaded_json,sort_keys=sort_key,indent=indent)
@click.command()
@click.argument('path',type=click.Path(exists=True))
@click.option('--sort','-s', is_flag=True)
def main (path,sort):
    # if no_sort:
    #     sort_key=True
    # else: sort_key=False

    with open (path,'r') as _:
        print (
            formater(_.read(),sort_key=sort)
              )
if __name__=="__main__":
    # parser=argparse.ArgumentParser(description="this is a json formater method")
    # parser.add_argument("--sort",action="store_true",help="set sorting")
    # args=parser.parse_args()
    # print(args.sort)
    #main(sys.argv[-1],no_sort=False)
    # main(sys.argv[-1])
    main()