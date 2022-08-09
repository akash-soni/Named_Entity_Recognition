echo [$(date)]: "START"
echo [$(date)]: "creating environment"
conda create --prefix ./env python=3.9 -y
echo [$(date)]: "activate environment"
source activate ./env
echo [$(date)]: "create folder and file structure"

for DIR in components config constants entity exception pipline utils
do 
    echo [$(date)]: "Creating", "ner/"$DIR 
    mkdir -p "ner/"$DIR # creating directories
    echo [$(date)]: "creating __init__.py inside above folders" # creating __init_.py in every folder
    touch "ner/"$DIR/"__init__.py"  # creating .py file for each stage
    
done

echo [$(date)]: "install requirements"
pip install -r requirements.txt
echo [$(date)]: "END"

# to remove everything -
# rm -rf env/ .gitignore conda.yaml README.md .git/