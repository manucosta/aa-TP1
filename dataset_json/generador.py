import json

ham_source = json.load(open('ham_source.json'))
spam_source = json.load(open('spam_source.json'))

print "Cargue source"

# Separo sets de testeo
ham_index = int(0.2 * len(ham_source))
data = ham_source[:ham_index]
with open('ham_test.json', 'w') as outfile:
  json.dump(data, outfile)

spam_index = int(0.2 * len(ham_source))
data = spam_source[:spam_index]
with open('spam_test.json', 'w') as outfile:
  json.dump(data, outfile)
print "Arme test"

# Separo sets de desarrollo
data = ham_source[ham_index:]
with open('ham_dev.json', 'w') as outfile:
  json.dump(data, outfile)

data = spam_source[spam_index:]
with open('spam_dev.json', 'w') as outfile:
  json.dump(data, outfile)
print "Arme dev"

# Separo sets de desarrollo reducido
ham_index = int(0.4 * len(ham_source))
data = ham_source[:ham_index]
with open('ham_dev_reducido.json', 'w') as outfile:
  json.dump(data, outfile)

spam_index = int(0.4 * len(spam_source))
data = spam_source[:spam_index]
with open('spam_dev_reducido.json', 'w') as outfile:
  json.dump(data, outfile)
print "Arme dev reducido"
