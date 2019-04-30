from waitress import serve
import flaskr
serve(flaskr.create_app(), listen='*:5000', threads=8)