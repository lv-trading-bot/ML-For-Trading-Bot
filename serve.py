from waitress import serve
import flaskr
serve(flaskr.create_app(), listen='*:3002', threads=8)