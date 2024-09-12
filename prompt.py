system = "A chat between a curious human and an artificial intelligence assistant. "
"The assistant gives helpful, detailed, and polite answers to the human's questions."





instruction = '''"You will be given a image-related question. Your job is to provide specific region of an image that you will attend to answer the question.
You must answer only by specific entity or object within the image, not the attributes or relations between objects.
Image will not be provided in this turn. 
Here are a few examples:

Question: What animal is drawn on the red notebook?
High-Quaility Region: red notebook

Question: Is the truck on the left or right side of the man with green shorts?
High-Quality Region: truck, man with green shorts


Now, based on the instruction above, provide high-quality region for the following question.
Do not answer the question.


Question: {query}
High-Quality Region:


'''

gqa_flickr30k_instruction = '''

Question: What is the color of the coat worn by the man?
Response: man

Question: Is the slide on the left or right side of the life buoy?
Response: slide, life buoy

Question: From the information on the black framed board, how long do we have to wait in line for this attraction?
Response: black framed board

Question: {qs}
Response: 
'''

region_annotation_instruction = '''
You are a region annotator. You will be given a image, and an entity name. Your job is to annotate whether the given image includes the given entity. 

If the given entity is present in the image, output
Always answer with Yes, or No.
'''