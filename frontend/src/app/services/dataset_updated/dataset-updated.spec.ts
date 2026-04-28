import { TestBed } from '@angular/core/testing';

import { DatasetUpdatedService } from './dataset-updated.service';

describe('DatasetUpdated', () => {
  let service: DatasetUpdatedService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(DatasetUpdatedService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
