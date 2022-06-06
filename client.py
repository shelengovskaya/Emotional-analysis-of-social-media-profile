import vk_api

USERNAME = # USERNAME
PASSWORD = # PASSWORD



class VKClient:
	def __init__(self, username=USERNAME, password=PASSWORD):

		vk_session = vk_api.VkApi(username, password)
		vk_session.auth()
		self.vk = vk_session.get_api()


	def get_posts(self, domain=None, count=10):


		if domain is not None:
			self.domain = domain

		posts = self.vk.wall.get(domain=self.domain, filter='owner', count=count)['items']

		texts = []
		links = []
		dates = []

		for post in posts[:min(len(posts), 10)]:
			if post['text']:
				texts.append(post['text'])
				links.append('https://vk.com/'+ self.domain+'?w=wall'+str(post['owner_id'])+'_'+str(post['id']))
				dates.append(post['date'])


		return texts, links, dates




# client = VKClient(USERNAME, PASSWORD)

# result = client.get_posts('mscyouth')

# print(result)