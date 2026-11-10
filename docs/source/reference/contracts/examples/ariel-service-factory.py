from osprey.services.ariel_search.capability import (
    close_ariel_service,
    get_ariel_search_service,
)


async def search_logbook():
    service = await get_ariel_search_service()
    return await service.search(query="RF cavity fault")


async def shutdown():
    await close_ariel_service()
