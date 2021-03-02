import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds']
creeds = ServiceAccountCredentials.from_json_keyfile_name('sheet.json', scope)
client = gspread.authorize(creeds)

sheet = client.open('checkexam').sheet1

demo = sheet.get_all_records()
print(demo)
