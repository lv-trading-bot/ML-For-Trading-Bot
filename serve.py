from waitress import serve
from config import Config
import flaskr
serve(flaskr.create_app(), listen='*:%s' % Config.PORT, threads=8)
