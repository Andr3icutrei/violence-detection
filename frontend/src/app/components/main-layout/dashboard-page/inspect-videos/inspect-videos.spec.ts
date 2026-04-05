import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InspectVideos } from './inspect-videos';

describe('InspectVideos', () => {
  let component: InspectVideos;
  let fixture: ComponentFixture<InspectVideos>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InspectVideos],
    }).compileComponents();

    fixture = TestBed.createComponent(InspectVideos);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
