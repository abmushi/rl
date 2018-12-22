import pygame
import constant as constant

pygame.init()
screen = pygame.display.set_mode((640, 480))
done = False

while not done:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			done = True

	#	draw
	color = (255,0,0,50)
	# pygame.draw.rect(screen, (0, 128, 255), pygame.Rect(30, 30, 30, 30))
	# pygame.display.flip()

	for i in range(0,64):
		for j in range(0,48):
			rect = pygame.Rect(i*10+1,j*10+1,10,10)
			screen.fill(color,rect)
			# pygame.draw.rect(screen, color, pygame.Rect(i*10+1,j*10+1,8,8))

	for i in range(0,int(constant._x/constant._unit)):
		pygame.draw.line(screen,(200,200,200),(i*constant._unit,0),(i*constant._unit,constant._y*constant._unit))
	for j in range(0,int(constant._y/constant._unit)):
		pygame.draw.line(screen,(200,200,200),(0,j*constant._unit),(constant._x*constant._unit,j*constant._unit))
			
	pygame.display.flip()