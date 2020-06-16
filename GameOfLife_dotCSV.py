import numpy as np
import pygame
import time

pygame.init()
width, height = 1000, 1000
screen = pygame.display.set_mode((height, width))

bg = 25, 25, 25
screen.fill(bg)

nxC, nyC = 100, 100

dimCW = int(width / nxC)
dimCH = int(height / nyC)

gameState = np.random.randint(2, size=(nxC, nyC))
pauseExec = False

while True:

    newGameState = np.copy(gameState)
    screen.fill(bg)
    time.sleep(0.01)

    ev = pygame.event.get()
    for event in ev:
        if event.type == pygame.KEYDOWN:
            pauseExec = not pauseExec

    mouseClick = pygame.mouse.get_pressed()
    if sum(mouseClick) > 0:
        posX, posY = pygame.mouse.get_pos()
        celX, celY = int(np.floor(posX / dimCW)), int(np.floor(posY / dimCH))
        newGameState[celX, celY] = 1  # not mouseClick[2]

    for y in range(0, nxC):
        for x in range(0, nyC):

            if not pauseExec:
                n_neigh = gameState[(x - 1) % nxC, (y - 1) % nyC] + \
                          gameState[x % nxC, (y - 1) % nyC] + \
                          gameState[(x + 1) % nxC, (y - 1) % nyC] + \
                          gameState[(x - 1) % nxC, y % nyC] + \
                          gameState[(x + 1) % nxC, y % nyC] + \
                          gameState[(x - 1) % nxC, (y + 1) % nyC] + \
                          gameState[x % nxC, (y + 1) % nyC] + \
                          gameState[(x + 1) % nxC, (y + 1) % nyC]

                if gameState[x, y] == 0 and n_neigh == 3:
                    newGameState[x, y] = 1
                elif gameState[x, y] == 1 and (n_neigh < 2 or n_neigh > 3):
                    newGameState[x, y] = 0

                poly = [(x * dimCW, y * dimCH),
                        ((x + 1) * dimCW, y * dimCH),
                        ((x + 1) * dimCW, (y + 1) * dimCH),
                        (x * dimCW, (y + 1) * dimCH)]

                if newGameState[x, y] == 0:
                    pygame.draw.polygon(screen, (128, 128, 128), poly, 1)
                else:
                    pygame.draw.polygon(screen, (128, 255, 128), poly, 0)

    gameState = np.copy(newGameState)

    pygame.display.flip()
