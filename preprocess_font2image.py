from os.path import join
import os
from collections import defaultdict
import argparse
import cPickle as pickle
from PIL import Image, ImageChops
from PIL import ImageDraw
from PIL import ImageFont
import pdb
import numpy as np
import time

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_image(img, path):
	new_path = '/'.join(path.split('/')[:-1])
	makedir(new_path)
	img.save(path)

def add_border(im, bg_color=(255,255,255)):
	new_size = (im.size[0]+4, im.size[1]+4)
	bordered_im = Image.new("RGB", new_size, bg_color)
	bordered_im.paste(im, (2,2))
	return bordered_im

def embed_image(img,w,h):
	img = add_border(img)
	img = trim(img)
	img_w, img_h = img.size
	maxsize = (w-2,h-2)
	if img_w > maxsize[0] or img_h > maxsize[1]:
		img.thumbnail(maxsize)
	img_w, img_h = img.size
	
	x = (w - img_w)/2
	y = (h - img_h)/2
	
	

	background = Image.new("RGBA", (w, h), (255, 255, 255))
	background.paste(img, (x,y))
	return background, img.size


def trim(im):
	bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
	diff = ImageChops.difference(im, bg)
	diff = ImageChops.add(diff, diff, 2.0, -100)
	bbox = diff.getbbox()
	if bbox:
		return im.crop(bbox)

def write_font(fnt, txt, w,h, target_size):
	temp_img = Image.new("RGBA", (1000,1000),(255,255,255))
	draw = ImageDraw.Draw(temp_img)
	text_x, text_y = fnt.getsize(txt)
	x = (1000 - text_x)/2
	y = (1000 - text_y)/2
	draw.text((x,y),txt,(0,0,0),font=fnt)
	trimmed_font_image = trim(temp_img)	
	trimmed_font_image.thumbnail(target_size)	
	trimmed_size = trimmed_font_image.size
	text_x = trimmed_size[0]
	text_y = trimmed_size[1]

	img = Image.new("RGBA",(w,h),(255,255,255))

	x = (w - text_x)/2
	y = (h - text_y)/2

	
	img.paste(trimmed_font_image, (x,y))
	return img


class Image_Container:
	#i might only need author, h, w, path, others just in case
	#h,w makes padding easier, as they are already known
	#path: for loading image during training (too many images to pre-load?)
	#author: for potential identification
	def __init__(self, author, h, w, word, form_id, image_path):
		self.image_path = image_path
		self.author = author
		self.form_id = form_id
		self.h = h
		self.w = w
		self.word = word

def process_words_file(file_path = 'words.txt', form_path = 'forms.txt'):
	word_dict = defaultdict(int)
	form_file = open(form_path)
	form_lines = [line.strip().split() for line in form_file]
	form_authors = {}
	print 'processing forms file'
	for line in form_lines:
		form_authors[line[0]] = line[1]
	print 'finished processing forms file, now processing words file'
	word_file = open(file_path)
	word_lines = [line.strip().split() for line in word_file]
	images = []
	max_h = 0
	max_w = 0
	for idx, line in enumerate(word_lines):
		if line[1] != 'ok':
			continue
		form_id = '-'.join(line[0].split('-')[:2])
		author = form_authors[form_id]
		image_path = join(form_id.split('-')[0], form_id, line[0]+'.png')
		w = int(line[5])
		h = int(line[6])
		word = ' '.join(line[8:])
		word_dict[word] += 1
		if w > max_w:
			max_w = w
		if h > max_h:
			max_h = h
		images.append(Image_Container(author, h, w, word, form_id, image_path))
	print max_w, max_h, 'max width, max height'

	return images, max_w, max_h, word_dict

def write_example_images(images, word_dict, max_width, max_height, size=100, write_path = 'typed_words/', font_path = '/home/johnzz/dan/zi2zi/HomemadeApple.ttf'):
	fnt = ImageFont.truetype(font_path,size)
	words_2_fontpath = {}
	print 'finding correct width'
	author_dicts = defaultdict(lambda: defaultdict(int))
	width = 0
	max_idx = 0
	for idx, word in enumerate(word_dict):
		txt = word

		text_x, text_y = fnt.getsize(txt)
		if text_x > width and idx != 5333:
			width = text_x
			max_idx = idx



	print str(max_idx) + ' is the widest image with a width of: ' + str(width+4)

	print 'now creating combined images'
	width = width + 4
	height = 256
	if width < max_w:
		width = max_w
	if height < max_h:
		height = max_h
	width = 256
	height = 64
	for idx, img_cont in enumerate(images):
		txt = img_cont.word
		h = img_cont.h
		w = img_cont.w
		label = img_cont.author
		if len(txt) < 10 and author_dicts[label][txt] == 0:
			author_dicts[label][txt] += 1
			img = Image.open('words/' + img_cont.image_path)
			example = Image.new("RGBA", (width * 2, height), (255, 255, 255))
			try:
				embedded_img, embedded_size = embed_image(img, width, height)
				example.paste(embedded_img, (0, 0))
				example.paste(write_font(fnt, txt, width, height, embedded_size), (width,0))	
				grey = example.convert('L')
				bw = grey.point(lambda x: 0 if x < 175 else 255, '1')
				example = Image.new("RGBA", (width * 2, height), (255, 255, 255))
				example.paste(bw, (0,0))
				make_image(example, join(write_path,label, label + '_' + txt + '_' + str(idx) + '.jpg'))
			except:
				print img_cont.image_path
				pass
		if idx%500 == 0:
			print 'processed ' + str(idx) + ' images'


parser = argparse.ArgumentParser(description='Processes images, converts font to images')
parser.add_argument('--src_font', dest='src_font', default = '/Users/main/generators/cangan/Mistral.ttf', help='path of the source font')
parser.add_argument('--char_size', dest='char_size', type=int, default=150, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--sample_dir', dest='sample_dir',default='typed_words/', help='directory to save examples')
parser.add_argument('--label', dest='label', type=int, default=0, help='label as the prefix of examples')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=20, help='x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=20, help='y_offset')
args = parser.parse_args()

if __name__ == "__main__":
	images, max_w, max_h, word_dict = process_words_file()
	#words2font = write_all_fonts(word_dict, args.x_offset, args.y_offset, args.char_size)
	write_example_images(images, word_dict, max_w, max_h)
	with open('images.pickle', 'w') as f:  
		pickle.dump([images, max_w, max_h, word_dict], f)
