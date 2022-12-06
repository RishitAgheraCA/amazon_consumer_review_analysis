from django.shortcuts import render
from django.views import View
from .services import scraper
import asyncio, re
from pyppeteer import launch

review_texts =[]
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


# Create your views here.
class InferenceView(View):
    def get(self, request):
        return render(request, 'scraping/amazon_index.html')

    def post(self, request):
        # print(request.POST)
        coroutine = scraper.scraper(request.POST.get('url'))
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            data = loop.run_until_complete(coroutine)
        finally:
            loop.close()

        sentiments = scraper.inference(data)
        print("view:", sentiments)
        perc_pos = (sentiments.count('Positive') / len(sentiments)) * 100

        # print(request)
        # import code
        # code.interact(local=dict(globals(), **locals()))
        # img = cv2.imdecode(np.fromstring(request.FILES['imagefile'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # predict = Inference()
        # if predict.image_classification(img):
        #     messege = 'Crop is Healthy'
        # else:
        #     messege = 'Crop is Diseased'
        messege = {'sentiments': sentiments,
                   'perc_pos': perc_pos}
        return render(request, 'scraping/amazon_index.html',messege)
