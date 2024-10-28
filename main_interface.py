import pandas as pd
import imp_setup as imps
from imp_setup import glob
import meta_funcs
import tfidf_sklearn
from fasthtml.common import *

# Sample DataFrame
df = pd.DataFrame()

app, rt = fast_app()

# Function to render the DataFrame as a table
def render_table(dataframe):
    headers = [Th(col) for col in dataframe.columns]
    rows = [
        Tr(*[Td(dataframe.iat[i, j]) for j in range(len(dataframe.columns))])
        for i in range(len(dataframe))
    ]
    return Table(
        Thead(Tr(*headers)),
        Tbody(*rows),
        cls="table"
    )

# Route to handle the search and print the search term to console
@rt('/search', methods=['POST'])
async def search(request):
    global df
    data = await request.form()
    search_term = data.get('search')
    # print(f"Search term: {search_term}")  # Prints the search term to the Python console
    df = tfidf_sklearn._fallback_(search_term)
    # render_table(df)
    return Redirect("/")

# Main page with search bar and table
@rt('/')
def get():
    return Div(
        H1("Data Table with Search"),
        
        # Search form
        Form(
            Input(type="text", name="search", placeholder="Search..."),
            Button("Submit", type="submit"),
            method="post",
            action="/search",
            cls="mb-3"
        ),
        
        render_table(df),
    )


print(glob.glob(f"*"))
df = pd.read_csv("datasets/teste_objects_description.csv")
render_table(df)
serve()
