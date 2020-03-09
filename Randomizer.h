#pragma once

#include <time.h>

//////////////////////////////////////////////////////////////////////
//
// Randomizer
//
// Ensures that the randomizer is seeded and provides a little more
// functionality to rand().
//
//
class CRandomizer
{
public:
	CRandomizer()
	{
		static bool isSeeded = false;

		if (!isSeeded)
		{
			srand((unsigned)time(NULL));
			isSeeded = true;
		}
	}

	static int random(int max)
	{
		return rand() % max;
	}

	static int randomSign()
	{
		if ((rand() & 0x0001))
			return -1;
		else
			return 1;
	}

	static bool randomFlip()
	{
		return !!(rand() & 0x0200);
	}

	static int rand1024()
	{
		return (rand() & 0x000003FF);
	}

	static int rand256()
	{
		return (rand() & 0x000000FF);
	}

	static double drand()
	{
		return((double)rand() / (double)RAND_MAX);
	}

	static float frand()
	{
		return((float)rand() / (float)RAND_MAX);
	}
};
