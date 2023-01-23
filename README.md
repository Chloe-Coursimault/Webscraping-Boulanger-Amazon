# Webscraping-Boulanger-Amazon
## Objectives of this project
With this project, we wish to help users getting good understanding of the laptops' market, before buying a new one. 

We will show users some tables and graphs to get them a good vision of products on Boulanger. We will also use Amazon reviews and descriptions on similar computers to display frequent words and recurrent themes appearing in those.

Disclaimer: Our project's aim is not to influence users but to give them an objective view on the market.

## Scraped websites
To conduct this project, we scraped 2 websites :
- Boulanger : Retrieve list of products sold by a French brand (price & caracteristics) -> ~960 products
- Amazon : Retrieve caracteristics, descriptions and reviews for similar products -> ~2700 products

## What do we propose?
- Tables : Products corresponding to a selection by the user, based on brand, size of RAM, OS and size of screen
- Graphs : Global views of prices and number of products for groups scpecified by the user
- Wordclouds : Most frequent words in reviews and descriptions for the selection
- Latent Dirichlet Allocation (LDA) : Recurrent themes for the selection

## Final interface
To present our results, we computed a UI using Dash with Python.
