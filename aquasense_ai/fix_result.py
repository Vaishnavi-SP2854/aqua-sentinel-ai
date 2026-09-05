
with open('templates/result.html', 'w', encoding='utf-8') as f:
    f.write(open('fix_result_content.txt', encoding='utf-8').read())
print('Done:', open('templates/result.html').read()[:50])

