# Custom_NER
Training a custom NER Model to accurately derive the target group of leads based on text data concerning their function within the company and belonging of department.


Within a CRM System, missing data is not uncommon. Therefore, this database had 50% missing data regarding the target group ('Head of IT' or 'Business Assistant'). In total there are 25 different choices for the target field. With a simple Bag-of-Words approach and Logistic Regression classifier I could achieve 75% balanced accuracy already. To improve the result, i build a custom NER Model on top, to have an ensemble model. The balanced accuracy rose by 10%.
